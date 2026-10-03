from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QRect, QSize, QStringListModel, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QMessageBox

from imagetagger.utils.annotations import sanitize_tag_text
from imagetagger.utils.io_utils import bg_write_text

if TYPE_CHECKING:
    from imagetagger.ui.main_window import MainWindow, TagListWidget, GlobalTagListWidget


class _BulkWriteSignaller(QObject):
    """Reports each finished background write (True when it succeeded) on the main thread."""

    written = pyqtSignal(bool)


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
        self._rename_thread: QThread | None = None
        self._rename_worker: TagPurgeWorker | None = None

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
        # Tags are stored normalized, as when added through the tag input; the
        # description keeps its case.
        if new_text and not self._window._is_description_like_annotation(new_text):
            new_text = sanitize_tag_text(new_text)

        duplicate = bool(new_text) and any(
            sanitize_tag_text(self._tag_list.item(i).text()) == sanitize_tag_text(new_text)
            for i in range(self._tag_list.count())
            if i != row
        )
        if not new_text or duplicate:
            removed = self._tag_list.takeItem(row)
            del removed
            self._update_tag_item_heights()
            self._window._sync_record_from_tag_list()
            if duplicate:
                self._window.statusBar().showMessage(f"Tag already exists: {new_text}")
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

        # Drop only the matching parts; descriptions and other tags stay as written.
        jobs: list[tuple] = []
        for record in affected:
            parts = self._window._split_record_annotations(record.text)
            new_text = self._window._serialize_tags(
                [part for part in parts if sanitize_tag_text(part) not in tags_set]
            )
            record.text = new_text
            jobs.append((record.text_path, new_text))

        # Refresh UI immediately so the user sees the change right away.
        self._rebuild_known_tags_from_records()
        self._refresh_tag_completions()
        self._refresh_changed_records(affected)

        self._write_texts_in_background(
            jobs,
            lambda done, total: f"Removing {tag_label} — writing {done} / {total}…",
            lambda total: f"Removed {tag_label} from {total} file(s).",
        )

    def _refresh_changed_records(self, changed: list) -> None:
        """Update list rows, and the tag list if the current image is among *changed*."""
        w = self._window
        for record in changed:
            index = w._record_index_for_image_path(record.image_path)
            if index >= 0:
                w._update_list_item_preview(index)
        current = w._current_record()
        if current is not None and current in changed:
            self._populate_tag_list(w._parse_annotations_for_tag_list(current.text))

    def _write_texts_in_background(self, jobs: list[tuple], progress_message, done_message) -> None:
        """Queue (text_path, text) writes in order behind any earlier saves, reporting progress.

        They are queued now, on the main thread, so a later edit of the same
        file is always written after them.  Failures are reported by the
        window's background write error handler.
        """
        w = self._window
        total = len(jobs)
        signaller = _BulkWriteSignaller(w)
        state = {"done": 0, "failed": 0}

        def on_written(ok: bool) -> None:
            state["done"] += 1
            if not ok:
                state["failed"] += 1
            if state["done"] < total:
                w.statusBar().showMessage(progress_message(state["done"], total))
                return
            message = done_message(total - state["failed"])
            if state["failed"]:
                message += f" {state['failed']} could not be saved."
            w.statusBar().showMessage(message)
            signaller.deleteLater()

        signaller.written.connect(on_written)
        w.statusBar().showMessage(progress_message(0, total))
        for path, text in jobs:
            bg_write_text(
                path,
                text,
                on_complete=lambda error: signaller.written.emit(error is None),
                durable=False,
            )

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

        changed = [record for record in affected if any(path == record.text_path for path, _ in jobs)]
        self._refresh_changed_records(changed)
        # Restore selection to the bumped tag at its new position.
        if self._window._current_record() in changed:
            for i in range(self._tag_list.count()):
                if self._tag_list.item(i).text().casefold() == tag_casefolded:
                    self._tag_list.setCurrentRow(i)
                    break

        tag_label = f'"{tag_to_bump}"'
        self._write_texts_in_background(
            jobs,
            lambda done, total: f"Bumping {tag_label} — writing {done} / {total}…",
            lambda total: f"Bumped {tag_label} in {total} file(s).",
        )

    def _rename_selected_tag(self) -> None:
        selected = self._known_tags_list.selectedItems()
        if len(selected) != 1:
            return

        old_tag = selected[0].data(Qt.ItemDataRole.UserRole) or selected[0].text().split(" (")[0]

        new_tag_raw, ok = QInputDialog.getText(
            self._window,
            "Rename tag",
            f'Rename "{old_tag}" to:',
            QLineEdit.EchoMode.Normal,
            old_tag,
        )
        if not ok:
            return

        new_tag = sanitize_tag_text(new_tag_raw)
        if not new_tag:
            QMessageBox.warning(self._window, "Rename tag", "Tag name cannot be empty.")
            return
        if new_tag == old_tag:
            return

        # Guard against renaming onto an already-existing tag.
        if new_tag in self._window.known_tags and new_tag != old_tag:
            confirm = QMessageBox(self._window)
            confirm.setWindowTitle("Rename tag")
            confirm.setText(
                f'Tag "{new_tag}" already exists.\n'
                "Renaming will merge all occurrences of the old tag into it.\n"
                "Continue?"
            )
            confirm.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            confirm.setDefaultButton(QMessageBox.StandardButton.No)
            confirm.raise_()
            confirm.activateWindow()
            if confirm.exec() != QMessageBox.StandardButton.Yes:
                return

        old_casefolded = old_tag.casefold()
        affected = [
            r for r in self._window.records
            if any(t.casefold() == old_casefolded for t in self._window._parse_tags(r.text))
        ]
        if not affected:
            return

        # Replace only the matching parts; descriptions and other tags stay as written.
        jobs: list[tuple] = []
        for record in affected:
            parts = self._window._split_record_annotations(record.text)
            renamed = [
                new_tag if sanitize_tag_text(part).casefold() == old_casefolded else part
                for part in parts
            ]
            # Deduplicate while preserving order (handles merge-into-existing case).
            seen: set[str] = set()
            deduped: list[str] = []
            for part in renamed:
                key = sanitize_tag_text(part).casefold()
                if key not in seen:
                    seen.add(key)
                    deduped.append(part)
            new_text = self._window._serialize_tags(deduped)
            record.text = new_text
            jobs.append((record.text_path, new_text))

        # Update in-memory tag index immediately.
        self._rebuild_known_tags_from_records()
        self._refresh_tag_completions()
        self._refresh_changed_records(affected)

        old_label = f'"{old_tag}"'
        new_label = f'"{new_tag}"'
        self._write_texts_in_background(
            jobs,
            lambda done, total: f"Renaming {old_label} → {new_label} — writing {done} / {total}…",
            lambda total: f"Renamed {old_label} → {new_label} in {total} file(s).",
        )
