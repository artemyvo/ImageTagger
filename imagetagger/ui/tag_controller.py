from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from PyQt6.QtCore import QRect, QSize, QStringListModel, QThread, Qt, QTimer
from PyQt6.QtWidgets import QLineEdit, QListWidget, QListWidgetItem, QMessageBox

from imagetagger.utils.annotations import sanitize_tag_text
from imagetagger.ui.workers import TagPurgeWorker

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow, TagListWidget, GlobalTagListWidget


class TagController:
    """Owns all tag-list and known-tags logic extracted from MainWindow (step 4.1).

    Receives the relevant widgets via constructor injection so it can be unit-tested
    in isolation. All reads/writes to shared MainWindow state go through
    ``self._window``.
    """

    def __init__(
        self,
        window: "MainWindow",
        tag_input: "QLineEdit",
        tag_list: "TagListWidget",
        tag_suggestions_model: "QStringListModel",
        known_tags_list: "GlobalTagListWidget",
        known_tags_filter: "QLineEdit",
    ) -> None:
        self._window = window
        self._tag_input = tag_input
        self._tag_list = tag_list
        self._tag_suggestions_model = tag_suggestions_model
        self._known_tags_list = known_tags_list
        self._known_tags_filter = known_tags_filter

        # Private state that previously lived on MainWindow but is only used here.
        self._updating_tag_list: bool = False
        self._purge_thread: QThread | None = None
        self._purge_worker: TagPurgeWorker | None = None
        self._bump_thread: QThread | None = None
        self._bump_worker: TagPurgeWorker | None = None

        # Debounce timer for the expensive known-tags sidebar rebuild.
        # Coalesces rapid consecutive calls (e.g. during merge-dialog navigation)
        # so only the final call actually rebuilds the list widget.
        self._known_tags_refresh_timer = QTimer()
        self._known_tags_refresh_timer.setSingleShot(True)
        self._known_tags_refresh_timer.setInterval(150)
        self._known_tags_refresh_timer.timeout.connect(self._do_refresh_known_tags_list)

    # ------------------------------------------------------------------
    # Tag-list display helpers
    # ------------------------------------------------------------------

    def _populate_tag_list(self, tags: list[str]) -> None:
        self._updating_tag_list = True
        self._tag_list.clear()
        for tag in tags:
            item = QListWidgetItem(tag)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._tag_list.addItem(item)
        self._update_tag_item_heights()
        self._updating_tag_list = False

    def _update_tag_item_heights(self) -> None:
        viewport_width = max(60, self._tag_list.viewport().width() - 10)
        fm = self._tag_list.fontMetrics()
        for i in range(self._tag_list.count()):
            item = self._tag_list.item(i)
            text_rect = fm.boundingRect(
                QRect(0, 0, viewport_width, 10000),
                int(Qt.TextFlag.TextWordWrap),
                item.text(),
            )
            item.setSizeHint(QSize(viewport_width, max(24, text_rect.height() + 8)))

    # ------------------------------------------------------------------
    # Tag-list event handlers
    # ------------------------------------------------------------------

    def _on_tag_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating_tag_list:
            return

        new_text = item.text().strip()
        row = self._tag_list.row(item)

        if not new_text:
            removed = self._tag_list.takeItem(row)
            del removed
            self._update_tag_item_heights()
            self._window._sync_record_from_tag_list()
            return

        self._updating_tag_list = True
        item.setText(new_text)
        self._updating_tag_list = False
        self._update_tag_item_heights()
        self._window._sync_record_from_tag_list()

    def _on_tags_reordered(self) -> None:
        self._update_tag_item_heights()
        self._window._sync_record_from_tag_list()

    def _remove_selected_tags(self) -> None:
        if self._window.current_index < 0 or self._window.current_index >= len(self._window.records):
            return

        selected = self._tag_list.selectedItems()
        if not selected:
            return

        first_row = min(self._tag_list.row(item) for item in selected)

        for item in selected:
            row = self._tag_list.row(item)
            removed = self._tag_list.takeItem(row)
            del removed

        self._update_tag_item_heights()
        self._window._sync_record_from_tag_list()

        count = self._tag_list.count()
        if count > 0:
            next_row = min(first_row, count - 1)
            self._tag_list.setCurrentRow(next_row)

    def _add_tag_from_input(self) -> None:
        if self._window.current_index < 0 or self._window.current_index >= len(self._window.records):
            return

        new_tag = sanitize_tag_text(self._tag_input.text())
        if not new_tag:
            return

        existing_keys = {
            sanitize_tag_text(existing_tag)
            for existing_tag in self._window._current_tags()
            if sanitize_tag_text(existing_tag)
        }
        if new_tag in existing_keys:
            self._window.statusBar().showMessage(f"Tag already exists: {new_tag}")
            self._tag_input.selectAll()
            return

        item = QListWidgetItem(new_tag)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._tag_list.addItem(item)
        self._tag_input.clear()
        QTimer.singleShot(0, self._tag_input.clear)
        self._update_tag_item_heights()
        self._window._sync_record_from_tag_list()

    # ------------------------------------------------------------------
    # Known-tags / completion helpers
    # ------------------------------------------------------------------

    def _rebuild_known_tags_from_records(self) -> None:
        counts: Counter[str] = Counter()
        for record in self._window.records:
            counts.update(self._window._parse_tags(record.text))
        self._window.tag_counts = counts
        self._window.known_tags = set(counts)

    def _update_tag_counts_incremental(self, old_tags: list[str], new_tags: list[str]) -> None:
        """Update tag_counts/known_tags for a single record change in O(|old|+|new|).

        Avoids the O(n × avg_tags) full rebuild triggered by
        ``_rebuild_known_tags_from_records`` when only one record changed.
        """
        w = self._window
        for tag in old_tags:
            new_count = w.tag_counts.get(tag, 0) - 1
            if new_count <= 0:
                try:
                    del w.tag_counts[tag]
                except KeyError:
                    pass
                w.known_tags.discard(tag)
            else:
                w.tag_counts[tag] = new_count
        for tag in new_tags:
            w.tag_counts[tag] = w.tag_counts.get(tag, 0) + 1
            w.known_tags.add(tag)

    def _sorted_tag_suggestions(self) -> list[str]:
        return sorted(
            self._window.known_tags,
            key=lambda tag: (-self._window.tag_counts.get(tag, 0), tag.lower(), tag),
        )

    def _refresh_tag_completions(self) -> None:
        suggestions = self._sorted_tag_suggestions()
        self._tag_suggestions_model.setStringList(suggestions)
        # Debounce the expensive sidebar rebuild; the autocomplete model above
        # is updated immediately so tag-input suggestions stay current.
        self._known_tags_refresh_timer.start()

    def _refresh_known_tags_list(self) -> None:
        """Schedule a debounced rebuild of the known-tags sidebar list."""
        self._known_tags_refresh_timer.start()

    def _do_refresh_known_tags_list(self) -> None:
        """Immediately rebuild the known-tags sidebar list. Called by the debounce timer."""
        filter_text = self._known_tags_filter.text().strip().casefold()

        self._known_tags_list.setUpdatesEnabled(False)
        try:
            self._known_tags_list.clear()
            for tag in self._sorted_tag_suggestions():
                if filter_text and filter_text not in tag.casefold():
                    continue
                count = self._window.tag_counts.get(tag, 0)
                item = QListWidgetItem(f"{tag} ({count})")
                item.setData(Qt.ItemDataRole.UserRole, tag)
                self._known_tags_list.addItem(item)
        finally:
            self._known_tags_list.setUpdatesEnabled(True)

    # ------------------------------------------------------------------
    # Global tag operations (delete / bump)
    # ------------------------------------------------------------------

    def _delete_global_tag(self, tags_to_delete: list[str]) -> None:
        tags_set = set(tags_to_delete)
        affected = [
            r for r in self._window.records
            if tags_set & set(self._window._parse_tags(r.text))
        ]
        if not affected:
            return

        tag_count = len(tags_set)
        tag_label = f'"{next(iter(tags_set))}"' if tag_count == 1 else f"{tag_count} tags"

        confirm = QMessageBox(self._window)
        confirm.setWindowTitle("Remove tags from all images")
        confirm.setText(f'Remove {tag_label} from {len(affected)} image(s)?')
        confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        confirm.setDefaultButton(QMessageBox.StandardButton.No)
        confirm.raise_()
        confirm.activateWindow()
        if confirm.exec() != QMessageBox.StandardButton.Yes:
            return

        # Compute new text for every affected record on the main thread (fast, no I/O).
        jobs: list[tuple] = []
        for record in affected:
            new_tags = [t for t in self._window._parse_tags(record.text) if t not in tags_set]
            new_text = self._window._serialize_tags(new_tags)
            record.text = new_text
            jobs.append((record.text_path, new_text))

        # Refresh UI immediately so the user sees the change right away.
        self._rebuild_known_tags_from_records()
        self._refresh_tag_completions()
        idx = self._window.current_index
        if 0 <= idx < len(self._window.records):
            record = self._window.records[idx]
            if record in affected:
                self._populate_tag_list(self._window._parse_tags(record.text))

        total = len(jobs)
        self._window.statusBar().showMessage(f'Removing {tag_label} — writing 0 / {total}…')

        worker = TagPurgeWorker(jobs)
        thread = QThread(self._window)
        worker.moveToThread(thread)

        def on_progress(done: int, total: int) -> None:
            self._window.statusBar().showMessage(f'Removing {tag_label} — writing {done} / {total}…')

        def on_finished() -> None:
            self._window.statusBar().showMessage(f'Removed {tag_label} from {total} file(s).')
            thread.quit()

        def on_failed(msg: str) -> None:
            QMessageBox.critical(self._window, "Save failed", f"Could not write some files:\n{msg}")
            thread.quit()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # Keep references so Python doesn't GC the thread/worker before they finish.
        self._purge_thread = thread
        self._purge_worker = worker

        thread.start()

    def _bump_selected_tag(self) -> None:
        if self._window.current_index < 0 or self._window.current_index >= len(self._window.records):
            return

        selected = self._known_tags_list.selectedItems()
        if len(selected) != 1:
            return

        tag_to_bump = selected[0].data(Qt.ItemDataRole.UserRole) or selected[0].text().split(" (")[0]
        tag_casefolded = tag_to_bump.casefold()

        affected = [
            r for r in self._window.records
            if any(
                t.casefold() == tag_casefolded
                for t in self._window._parse_annotations_for_tag_list(r.text)
            )
        ]
        if not affected:
            return

        jobs: list[tuple] = []
        for record in affected:
            parsed = self._window._parse_annotations_for_tag_list(record.text)
            has_description = bool(parsed) and self._window._is_description_like_annotation(parsed[0])
            insert_pos = 1 if has_description else 0

            current_pos = next(
                (i for i, t in enumerate(parsed) if t.casefold() == tag_casefolded), None
            )
            if current_pos is None or current_pos == insert_pos:
                continue

            new_tags = [t for t in parsed if t.casefold() != tag_casefolded]
            new_tags.insert(insert_pos, tag_to_bump)
            new_text = self._window._serialize_tags(new_tags)
            record.text = new_text
            jobs.append((record.text_path, new_text))

        if not jobs:
            self._window.statusBar().showMessage(
                f'"{tag_to_bump}" is already at the top in all affected images.'
            )
            return

        # Refresh the current record's tag list immediately.
        current_record = self._window.records[self._window.current_index]
        if current_record in affected:
            self._populate_tag_list(
                self._window._parse_annotations_for_tag_list(current_record.text)
            )
            # Restore selection to the bumped tag at its new position.
            for i in range(self._tag_list.count()):
                if self._tag_list.item(i).text().casefold() == tag_casefolded:
                    self._tag_list.setCurrentRow(i)
                    break

        total = len(jobs)
        tag_label = f'"{tag_to_bump}"'
        self._window.statusBar().showMessage(f'Bumping {tag_label} — writing 0 / {total}…')

        worker = TagPurgeWorker(jobs)
        thread = QThread(self._window)
        worker.moveToThread(thread)

        def on_progress(done: int, total: int = total) -> None:
            self._window.statusBar().showMessage(f'Bumping {tag_label} — writing {done} / {total}…')

        def on_finished(total: int = total) -> None:
            self._window.statusBar().showMessage(f'Bumped {tag_label} in {total} file(s).')
            thread.quit()

        def on_failed(msg: str) -> None:
            QMessageBox.critical(self._window, "Save failed", f"Could not write some files:\n{msg}")
            thread.quit()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._bump_thread = thread
        self._bump_worker = worker
        thread.start()
